/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.cassandra.index.sai.disk.v2.blockindex;

import java.io.Closeable;
import java.io.IOException;

import org.apache.cassandra.index.sai.disk.v1.DirectReaders;
import org.apache.cassandra.index.sai.disk.v1.LeafOrderMap;
import org.apache.cassandra.index.sai.disk.v2.PrefixBytesReader;
import org.apache.cassandra.index.sai.utils.SeekingRandomAccessInput;
import org.apache.cassandra.index.sai.utils.SharedIndexInput;
import org.apache.cassandra.index.sai.utils.SharedIndexInput2;
import org.apache.cassandra.io.util.FileUtils;
import org.apache.lucene.util.BytesRef;

import static org.apache.cassandra.index.sai.disk.v2.blockindex.PrefixBlockWriter.INDEX_INTERVAL;

public class PrefixBlockReader implements BlockValuesDeserializer, Closeable
{
    final SeekingRandomAccessInput seekInput;
    final SharedIndexInput input;
    final SharedIndexInput2 upperTermsInput, lowerTermsInput;

    byte upperCount;
    byte lastBlockCount;
    PrefixBytesReader upperTermsReader;
    PrefixBytesReader lowerTermsReader;
    private int idx;
    private BytesRef upperTerm;
    private long currentFP = -1;
    private long lowerTermsStartFP;
    private long currentLowerTermsFP;
    DirectReaders.Reader reader;
    long lowerBlockSizeDeltasFP;

    public PrefixBlockReader(final SharedIndexInput input) throws IOException
    {
        this.input = input;
        upperTermsInput = new SharedIndexInput2(input.sharedCopy());
        lowerTermsInput = new SharedIndexInput2(input.sharedCopy());
        seekInput = new SeekingRandomAccessInput(input.sharedCopy());
    }

    public void reset(final long fp) throws IOException
    {
        input.seek(fp);
        upperCount = input.readByte();
        lastBlockCount = input.readByte();
        final short upperTermsSize = input.readShort();
        final short lowerBlockSizeDeltasSize = input.readShort();

        final long fp2 = input.getFilePointer();

        upperTermsReader = new PrefixBytesReader(upperTermsInput);
        upperTermsReader.reset(fp2);

        input.seek(fp2 + upperTermsSize); // seek to the lowerBlockSizeDeltas

        final int bits = input.readByte();

        lowerBlockSizeDeltasFP = input.getFilePointer();

        reader = DirectReaders.getReaderForBitsPerValue((byte)bits);

        lowerTermsStartFP = currentLowerTermsFP = fp2 + lowerBlockSizeDeltasSize + upperTermsSize;
    }

    @Override
    public int ordinal() throws IOException
    {
        return idx;
    }

    @Override
    public void close() throws IOException
    {
        FileUtils.close(upperTermsInput, lowerTermsInput, seekInput);
    }

    public BytesRef next() throws IOException
    {
        final int upperIdx = idx / INDEX_INTERVAL;
        final int lowerIdx = idx % INDEX_INTERVAL;

        if (upperIdx == upperCount - 1 && lowerIdx == lastBlockCount)
        {
            return null;
        }

        if (idx % INDEX_INTERVAL == 0)
        {
            upperTerm = upperTermsReader.next();

            // TODO: reuse lowerTermsReader
            lowerTermsReader = new PrefixBytesReader(lowerTermsInput);
            lowerTermsReader.reset(currentLowerTermsFP);

            final long lowerBlockSize = LeafOrderMap.getValue(seekInput, lowerBlockSizeDeltasFP, upperIdx, reader);
            currentLowerTermsFP += lowerBlockSize;
        }
        idx++;
        return lowerTermsReader.next();
    }

    public int getUpperOrdinal()
    {
        return upperTermsReader.getOrdinal();
    }

    public BytesRef getCurrentUpperTerm()
    {
         return upperTermsReader.current();
    }

    public void initLowerTerms(int upperIdx) throws IOException
    {
        currentLowerTermsFP = lowerTermsStartFP;
        // iterate over the bytes lengths of each lower terms block
        // to arrive at a file pointer
        for (int x = 0; x < upperIdx; x++)
        {
            currentLowerTermsFP += getLowerTermsSizeBytes(x);
        }
        lowerTermsReader = new PrefixBytesReader(lowerTermsInput);
        lowerTermsReader.reset(currentLowerTermsFP);
    }

    public int getLowerTermsSizeBytes(int upperIdx)
    {
        return LeafOrderMap.getValue(seekInput, lowerBlockSizeDeltasFP, upperIdx, reader);
    }

    public BytesRef seek(BytesRef target) throws IOException
    {
        // TODO: optimize if the target is already in the current sub-block
        final BytesRef upperTerm = seekUpper(target);

        final int upperIdx = getUpperOrdinal() - 2;

        initLowerTerms(upperIdx);

        while (true)
        {
            final BytesRef lowerTerm = lowerTermsReader.next();
            if (lowerTerm == null)
            {
                this.upperTerm = upperTermsReader.next();
                return seek(target);
            }
            if (lowerTerm != null && target.compareTo(lowerTerm) <= 0)
            {
                return lowerTerm;
            }
        }
    }

    public BytesRef seekUpper(BytesRef target) throws IOException
    {
        if (upperTerm != null && target.compareTo(upperTerm) <= 0)
        {
            return upperTerm;
        }
        final BytesRef next = upperTermsReader.next();
        if (next != null)
        {
            upperTerm = next;
        }
        else
        {
            return upperTerm;
        }
        return seekUpper(target);
    }
}
